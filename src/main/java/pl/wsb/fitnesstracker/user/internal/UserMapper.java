package pl.wsb.fitnesstracker.user.internal;

/**
 * Mapper responsible for converting {@link User} entities
 * into User-related Data Transfer Objects (DTO).
 */


import org.springframework.stereotype.Component;
import pl.wsb.fitnesstracker.user.api.User;
import pl.wsb.fitnesstracker.user.api.UserDto;
import pl.wsb.fitnesstracker.user.api.UserEmailDto;

/**
 * Mapper for converting User entities to various DTO representations.
 */
@Component
public class UserMapper {

    /**
     * Converts a {@link User} entity into a full {@link UserDto}.
     *
     * @param user the user entity to be converted
     * @return a {@link UserDto} containing complete user information
     */



    public UserDto toDto(User user) {
        return new UserDto(user.getId(),
                user.getFirstName(),
                user.getLastName(),
                user.getBirthdate(),
                user.getEmail());
    }

    /**
     * Converts a {@link User} entity into a {@link UserEmailDto}
     * containing only the user identifier and email address.
     *
     * @param user the user entity to be converted
     * @return a {@link UserEmailDto} with limited user data
     */

    UserEmailDto toEmailDto(User user) {
        return new UserEmailDto(user.getId(), user.getEmail());
    }

}
